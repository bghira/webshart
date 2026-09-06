use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use serde_json::Value;

pub(crate) fn validate_field_names(old_name: &str, new_name: &str) -> PyResult<()> {
    if old_name.trim().is_empty() || new_name.trim().is_empty() {
        return Err(PyValueError::new_err("Field names must not be empty"));
    }
    if old_name == new_name {
        return Ok(());
    }
    for name in [old_name, new_name] {
        if matches!(
            name,
            "path"
                | "fname"
                | "filename"
                | "offset"
                | "length"
                | "size"
                | "json_path"
                | "json_offset"
                | "json_length"
        ) {
            return Err(PyValueError::new_err(format!(
                "Cannot rename structural index field '{name}'"
            )));
        }
    }
    Ok(())
}

pub(crate) fn rename_fields(
    metadata: &mut Value,
    old_name: &str,
    new_name: &str,
    overwrite: bool,
    shard_name: &str,
) -> PyResult<usize> {
    let entries: Box<dyn Iterator<Item = &mut Value> + '_> = match metadata.get_mut("files") {
        Some(Value::Object(files)) => Box::new(files.values_mut()),
        Some(Value::Array(files)) => Box::new(files.iter_mut()),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Shard '{shard_name}' metadata must contain a 'files' dict or list"
            )));
        }
    };
    let mut updated = 0;
    for entry in entries {
        let entry = entry.as_object_mut().ok_or_else(|| {
            PyValueError::new_err(format!("Invalid file entry in shard '{shard_name}'"))
        })?;
        if !entry.contains_key(old_name) {
            continue;
        }
        if !overwrite && entry.contains_key(new_name) {
            return Err(PyValueError::new_err(format!(
                "Field '{new_name}' already exists in shard '{shard_name}'; use overwrite=True to replace it"
            )));
        }
        if let Some(value) = entry.remove(old_name) {
            entry.insert(new_name.to_string(), value);
            updated += 1;
        }
    }
    Ok(updated)
}
